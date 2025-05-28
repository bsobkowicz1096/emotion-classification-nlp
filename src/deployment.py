#!/usr/bin/env python3
"""
Wdrożenie modelu klasyfikacji emocji
"""

import pandas as pd
import numpy as np
import pickle
import re
from sklearn.metrics import classification_report
import os

class EmotionPredictor:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.stop_words = None
        
    def load_model(self, model_path='results/emotion_model.pkl'):
        """Wczytanie wytrenowanego modelu"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.vectorizer = model_data['vectorizer']
                self.stop_words = model_data['stop_words']
            print(f"Model wczytany z: {model_path}")
        except FileNotFoundError:
            print(f"Nie znaleziono pliku modelu: {model_path}")
            print("Najpierw wytrenuj model używając notebook'a")
    
    def save_model(self, model, vectorizer, stop_words, model_path='results/emotion_model.pkl'):
        """Zapisanie modelu do pliku"""
        os.makedirs('results', exist_ok=True)
        
        model_data = {
            'model': model,
            'vectorizer': vectorizer,
            'stop_words': stop_words
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def preprocess_text(self, text):
        """Przetwarzanie tekstu (identyczne jak w treningu)"""
        if pd.isna(text):
            return ""
        
        # Małe litery i czyszczenie
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenizacja i filtrowanie stop words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        words = [word for word in words if word not in self.stop_words]
        
        return ' '.join(words)
    
    def predict_emotion(self, text):
        """Predykcja emocji dla pojedynczego tekstu"""
        if self.model is None:
            raise ValueError("Model nie został wczytany. Użyj load_model() lub save_model()")
        
        # Przetwarzanie tekstu
        processed_text = self.preprocess_text(text)
        
        if not processed_text:
            return "unknown", 0.0
        
        # Wektoryzacja
        text_vector = self.vectorizer.transform([processed_text])
        
        # Predykcja
        prediction = self.model.predict(text_vector)[0]
        probabilities = self.model.predict_proba(text_vector)[0]
        confidence = max(probabilities)
        
        return prediction, confidence
    
    def predict_batch(self, texts):
        """Predykcja dla listy tekstów"""
        results = []
        for text in texts:
            emotion, confidence = self.predict_emotion(text)
            results.append({
                'text': text,
                'predicted_emotion': emotion,
                'confidence': confidence
            })
        return results

def demonstrate_predictions(test_df, y_test, y_pred, y_proba):
    """Demonstracja predykcji na wybranych przypadkach"""
    print("=== DEMONSTRACJA WDROŻONEGO MODELU ===")
    
    # Przygotowanie przykładów
    examples = []
    
    # 1. Wysokie prawdopodobieństwo (model pewny)
    confident_mask = y_proba.max(axis=1) > 0.9
    confident_indices = np.where(confident_mask)[0]
    if len(confident_indices) >= 2:
        examples.extend(confident_indices[:2])
    
    # 2. Niepewność modelu
    uncertain_mask = (y_proba.max(axis=1) < 0.7) & (y_proba.max(axis=1) > 0.4)
    uncertain_indices = np.where(uncertain_mask)[0]
    if len(uncertain_indices) >= 2:
        examples.extend(uncertain_indices[:2])
    
    # 3. Błędy modelu
    error_mask = y_test != y_pred
    error_indices = np.where(error_mask)[0]
    if len(error_indices) >= 2:
        examples.extend(error_indices[:2])
    
    # Demonstracja predykcji
    predictor = EmotionPredictor()
    try:
        predictor.load_model()
        
        print(f"\nPrzykłady predykcji (wybrano {len(examples)} przypadki/przypadków):")
        print("-" * 80)
        
        for i, idx in enumerate(examples[:6]):
            original_text = test_df.iloc[idx]['text']
            true_emotion = y_test[idx]
            pred_emotion = y_pred[idx]
            confidence = y_proba[idx].max()
            
            print(f"\n{i+1}. Przykład:")
            print(f"   Tekst: '{original_text[:80]}{'...' if len(original_text) > 80 else ''}'")
            print(f"   Prawdziwa emocja: {true_emotion}")
            print(f"   Predykcja modelu: {pred_emotion} (pewność: {confidence:.3f})")
            
            if true_emotion == pred_emotion:
                print("   ✓ Poprawna klasyfikacja")
            else:
                print("   ✗ Błędna klasyfikacja")
                
    except Exception as e:
        print(f"Błąd podczas demonstracji: {e}")
        print("Upewnij się, że model został zapisany w poprzednich krokach")

def deploy_model(best_model, vectorizer, stop_words, test_df, y_test, y_pred, y_proba):
    """Główna funkcja wdrożenia"""
    
    # Zapisanie modelu
    predictor = EmotionPredictor()
    predictor.save_model(best_model, vectorizer, stop_words)
    
    # Demonstracja
    demonstrate_predictions(test_df, y_test, y_pred, y_proba)
    
    return predictor

if __name__ == "__main__":
    main()