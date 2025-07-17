import joblib
import os
from sklearn.metrics import accuracy_score


class OldGenreClassifier:
    """
    Original logistic regression model classifier
    """
    
    def __init__(self, model_dir='Model'):
        self.model_dir = model_dir
        self.model_pipeline = None
        self.label_to_genre = None
        self.is_loaded = False
        self.model_info = {
            'name': 'Logistic Regression Pipeline',
            'type': 'TF-IDF + Logistic Regression',
            'accuracy': '~65-70%',
            'features': 'TF-IDF Vectorization'
        }
    
    def load_models(self):
        """
        Load the original logistic regression model
        """
        try:
            model_pipeline_path = os.path.join(self.model_dir, 'logistic_regression_pipeline.joblib')
            label_path = os.path.join(self.model_dir, 'label_to_genre.joblib')
            
            if not os.path.exists(model_pipeline_path) or not os.path.exists(label_path):
                print(f"Old model files not found in {self.model_dir}")
                return False
                
            self.model_pipeline = joblib.load(model_pipeline_path)
            self.label_to_genre = joblib.load(label_path)
            
            self.is_loaded = True
            print("Old logistic regression model loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading old model: {e}")
            self.is_loaded = False
            return False
    
    def predict(self, lyrics_text):
        """
        Predict genre using original model
        """
        if not self.is_loaded:
            raise ValueError("Old model not loaded. Call load_models() first.")
        
        try:
            # Make prediction using pipeline
            predicted_label_index = self.model_pipeline.predict([lyrics_text])[0]
            predicted_genre = self.label_to_genre[predicted_label_index]
            
            # Get probabilities
            probabilities = self.model_pipeline.predict_proba([lyrics_text])[0]
            
            # Convert to genre probabilities
            genre_probabilities = {
                self.label_to_genre[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            }
            
            confidence = max(probabilities)
            
            return {
                'predicted_genre': predicted_genre,
                'confidence': float(confidence),
                'genre_probabilities': genre_probabilities,
                'model_info': self.model_info
            }
            
        except Exception as e:
            print(f"Error in old model prediction: {e}")
            raise
