#!/usr/bin/env python3
"""
Trenowanie i porównanie modeli klasyfikacji emocji
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

class EmotionModels:
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def initialize_models(self):
        """Inicjalizacja modeli z odpowiednimi parametrami"""
        self.models = {
            'naive_bayes': MultinomialNB(alpha=1.0),
            'logistic_regression': LogisticRegression(
                max_iter=1000, 
                random_state=42,
                class_weight='balanced'  # dla nierównowagi klas
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
        }
        
    
    def evaluate_model(self, name, model, X_val, y_val):
        """Ewaluacja modelu na zbiorze walidacyjnym"""
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        # Szczegółowy raport
        report = classification_report(y_val, y_pred, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'classification_report': report
        }
    
    def train_all_models(self, X_train, y_train):
        """Trenowanie wszystkich modeli"""
        print("=== TRENOWANIE MODELI ===")
        
        self.initialize_models()
        
        for name, model in self.models.items():
            print(f"Trenowanie {name}...")
            model.fit(X_train, y_train)
            
    def evaluate_all_models(self, X_train, y_train, X_val, y_val):
        """Ewaluacja wszystkich modeli"""
        print("\n=== EWALUACJA NA ZBIORZE WALIDACYJNYM ===")
        
        for name, model in self.models.items():
            # Accuracy na train
            train_accuracy = model.score(X_train, y_train)
            
            # Ewaluacja na val
            results = self.evaluate_model(name, model, X_val, y_val)
            results['train_accuracy'] = train_accuracy
            
            self.results[name] = results
        
        return self.results
    
    def compare_models(self):
        """Porównanie wyników modeli"""
        print("\n=== PORÓWNANIE MODELI ===")
        
        comparison = []
        for name, results in self.results.items():
            val_acc = results['accuracy']
            train_acc = results['train_accuracy']
            delta = train_acc - val_acc
            
            comparison.append({
                'Model': name.replace('_', ' ').title(),
                'Train Acc': train_acc,
                'Val Acc': val_acc,
                'Delta': delta,
                'Weighted F1': results['classification_report']['weighted avg']['f1-score']
            })
        
        # Sortowanie po val accuracy
        comparison = sorted(comparison, key=lambda x: x['Val Acc'], reverse=True)
        
        print(f"{'Model':<20} {'Train Acc':<10} {'Val Acc':<10} {'Delta':<8} {'Weighted F1':<12}")
        print("-" * 60)
        for row in comparison:
            print(f"{row['Model']:<20} {row['Train Acc']:<10.4f} {row['Val Acc']:<10.4f} {row['Delta']:<8.4f} {row['Weighted F1']:<12.4f}")
        
        # Najlepszy model
        best_model_name = comparison[0]['Model'].lower().replace(' ', '_')
        print(f"\nNajlepszy model: {comparison[0]['Model']} (Val Accuracy: {comparison[0]['Val Acc']:.4f})")
        
        return best_model_name, comparison
    
    def get_f1_scores_by_emotion(self):
        """Tabela F1 scores: modele x emocje"""
        print("\n=== F1 SCORES PER EMOCJA ===")
        
        emotions = ['sadness', 'anger', 'love', 'surprise', 'fear', 'joy']
        
        # Header
        header = f"{'Model':<20}"
        for emotion in emotions:
            header += f"{emotion.capitalize():<10}"
        print(header)
        print("-" * (20 + 10 * len(emotions)))
        
        # Wiersze z wynikami
        for name, results in self.results.items():
            row = f"{name.replace('_', ' ').title():<20}"
            report = results['classification_report']
            
            for emotion in emotions:
                if emotion in report:
                    f1_score = report[emotion]['f1-score']
                    row += f"{f1_score:<10.3f}"
                else:
                    row += f"{'N/A':<10}"
            
            print(row)

def train_and_compare_models(X_train, y_train, X_val, y_val):
    """Główna funkcja trenowania i porównania modeli"""
    
    emotion_models = EmotionModels()
    
    # Trenowanie
    emotion_models.train_all_models(X_train, y_train)
    
    # Ewaluacja
    results = emotion_models.evaluate_all_models(X_train, y_train, X_val, y_val)
    
    # Porównanie
    best_model_name, comparison = emotion_models.compare_models()
    
    # Tabela F1 per emocja
    emotion_models.get_f1_scores_by_emotion()
    
    return emotion_models, best_model_name, comparison

if __name__ == "__main__":
    main()