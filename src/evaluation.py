#!/usr/bin/env python3
"""
Dostrajanie hiperparametrów i finalna ewaluacja modelu
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class ModelEvaluator:
    def __init__(self):
        self.best_model = None
        self.grid_search = None
        
    def tune_logistic_regression(self, X_train, y_train, X_val, y_val):
        """Grid search dla Logistic Regression"""
        print("=== DOSTRAJANIE HIPERPARAMETRÓW ===")
        
        param_grid = {
            'C': [0.1, 0.5, 1.0, 2.0, 5.0],
            'solver': ['liblinear', 'lbfgs'],
            'max_iter': [1000, 2000]
        }
        
        self.grid_search = GridSearchCV(
            LogisticRegression(random_state=42, class_weight='balanced'),
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )
        
        self.grid_search.fit(X_train, y_train)
        self.best_model = self.grid_search.best_estimator_
        
        # Wyniki tuningu
        print(f"Najlepsze parametry: {self.grid_search.best_params_}")
        
        # Accuracy po tuningu
        train_acc = self.best_model.score(X_train, y_train)
        val_acc = self.best_model.score(X_val, y_val)
        
        print(f"Train accuracy: {train_acc:.4f}")
        print(f"Validation accuracy: {val_acc:.4f}")
        
        return self.best_model
    
    def final_evaluation(self, X_test, y_test):
        """Finalna ewaluacja na test set"""
        print("\n=== FINALNA EWALUACJA NA TEST SET ===")
        
        y_pred = self.best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        print("\nClassification Report:")
        print(f"{'Emotion':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 40)
        
        emotions = sorted([k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']])
        for emotion in emotions:
            metrics = report[emotion]
            print(f"{emotion:<10} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} {metrics['f1-score']:<10.3f}")
        
        # Weighted average
        weighted = report['weighted avg']
        print(" ")
        print(f"{'Weighted':<10} {weighted['precision']:<10.3f} {weighted['recall']:<10.3f} {weighted['f1-score']:<10.3f}")
        
        return y_pred, test_accuracy
    
    def plot_confusion_matrix(self, y_test, y_pred, save_plot=True):
        """Wizualizacja confusion matrix"""
        print("\n=== CONFUSION MATRIX ===")
        
        cm = confusion_matrix(y_test, y_pred)
        emotions = sorted(set(y_test))
        
        # Heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=emotions, yticklabels=emotions)
        plt.title('Confusion Matrix - Test Set')
        plt.ylabel('True Emotion')
        plt.xlabel('Predicted Emotion')
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def analyze_feature_importance(self, vectorizer, top_n=10):
        """Tabela top cech per emocja"""
        print(f"\n=== TOP {top_n} CECH PER EMOCJA ===")
        
        feature_names = vectorizer.get_feature_names_out()
        emotions = self.best_model.classes_
        
        # Header
        header = f"{'Rank':<5}"
        for emotion in emotions:
            header += f"{emotion.capitalize():<12}"
        print(header)
        print("-" * (5 + 12 * len(emotions)))
        
        # Top cechy per emocja
        top_features = {}
        for i, emotion in enumerate(emotions):
            coeffs = self.best_model.coef_[i]
            top_indices = coeffs.argsort()[-top_n:][::-1]
            top_features[emotion] = [feature_names[idx] for idx in top_indices]
        
        # Wiersze z cechami
        for rank in range(top_n):
            row = f"{rank+1:<5}"
            for emotion in emotions:
                feature = top_features[emotion][rank] if rank < len(top_features[emotion]) else ""
                row += f"{feature:<12}"
            print(row)
    
def evaluate_final_model(X_train, y_train, X_val, y_val, X_test, y_test, preprocessor, vectorizer):
    """Główna funkcja ewaluacji"""
    
    evaluator = ModelEvaluator()
    
    # Tuning
    best_model = evaluator.tune_logistic_regression(X_train, y_train, X_val, y_val)
    
    # Finalna ewaluacja
    y_pred, test_accuracy = evaluator.final_evaluation(X_test, y_test)
    
    # Confusion matrix
    cm = evaluator.plot_confusion_matrix(y_test, y_pred)
    
    # Feature importance
    evaluator.analyze_feature_importance(vectorizer)
    
    return evaluator, y_pred, test_accuracy

if __name__ == "__main__":
    main()