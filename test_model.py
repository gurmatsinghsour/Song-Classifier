#!/usr/bin/env python3
"""
Quick test script to validate the song genre classifier
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import GenreClassifier


def test_classifier():
    """Test the genre classifier with sample lyrics"""
    
    print("Testing Song Genre Classifier")
    print("=" * 40)
    
    # Initialize classifier
    classifier = GenreClassifier()
    classifier.load_models()
    
    if not classifier.is_loaded:
        print("ERROR: Failed to load models")
        return False
    
    print("Models loaded successfully")
    
    # Test cases
    test_cases = [
        ("Rock Sample", "Electric guitar screaming through the night, heavy drums pounding like thunder, rock and roll forever"),
        ("Country Sample", "Country roads, truck driving, hometown pride, simple life in the countryside"),
        ("R&B Sample", "Smooth vocals, soulful melody, rhythm and blues, heartfelt emotions"),
        ("Hip Hop Sample", "Beats drop hard, rap flow, hip hop style, urban sound and rhythm")
    ]
    
    print("\nTesting predictions:")
    print("-" * 40)
    
    all_passed = True
    
    for name, lyrics in test_cases:
        try:
            result = classifier.predict(lyrics)
            predicted_genre = result['predicted_genre']
            confidence = result['confidence']
            
            print(f"\n{name}:")
            print(f"  Predicted: {predicted_genre}")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Top 3 probabilities:")
            
            sorted_probs = sorted(result['genre_probabilities'].items(), 
                                key=lambda x: x[1], reverse=True)[:3]
            for genre, prob in sorted_probs:
                print(f"    {genre}: {prob:.3f}")
            
        except Exception as e:
            print(f"ERROR testing {name}: {e}")
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("SUCCESS: All tests passed!")
        print("The classifier is working correctly.")
    else:
        print("FAILURE: Some tests failed.")
    
    return all_passed


if __name__ == "__main__":
    success = test_classifier()
    sys.exit(0 if success else 1)
