#!/usr/bin/env python3
"""
Przetwarzanie danych tekstowych dla klasyfikacji emocji
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from src.data_exploration import find_generic_words

class TextPreprocessor:
    def __init__(self):
        self.vectorizer = None
        self.stop_words = set()
        
    def load_all_data(self, train_path, val_path, test_path):
        """Wczytanie wszystkich zbiorów danych"""
        def load_txt(file_path):
            texts, labels = [], []
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if line and ';' in line:
                        text, label = line.rsplit(';', 1)
                        texts.append(text)
                        labels.append(label)
            return pd.DataFrame({'text': texts, 'label': labels})
        
        self.train_df = load_txt(train_path)
        self.val_df = load_txt(val_path)
        self.test_df = load_txt(test_path)
        
        print(f"Wczytano dane: train={len(self.train_df)}, val={len(self.val_df)}, test={len(self.test_df)}")
        return self.train_df, self.val_df, self.test_df
    
    def detect_stop_words(self, df, threshold=0.67):
        """Automatyczne wykrywanie słów niecharakterystycznych"""
        return find_generic_words(df, threshold)
    
    def clean_text(self, text):
        """Podstawowe czyszczenie tekstu"""
        if pd.isna(text):
            return ""
        
        # Małe litery i usunięcie znaków specjalnych
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenizacja i filtrowanie
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        words = [word for word in words if word not in self.stop_words]
        
        return ' '.join(words)
    
    def prepare_features(self, max_features=5000, ngram_range=(1, 2)):
        """Przygotowanie cech TF-IDF"""
        print("Przygotowanie cech...")
        
        # Wykrycie stop words tylko na danych treningowych
        self.stop_words = self.detect_stop_words(self.train_df)
        print(f"Wykryte stop words: {len(self.stop_words)} słów")
        
        # Przetwarzanie tekstów
        print("Przetwarzanie tekstów...")
        self.train_df['processed'] = self.train_df['text'].apply(self.clean_text)
        self.val_df['processed'] = self.val_df['text'].apply(self.clean_text)
        self.test_df['processed'] = self.test_df['text'].apply(self.clean_text)
        
        # Usunięcie pustych tekstów
        self.train_df = self.train_df[self.train_df['processed'].str.len() > 0]
        self.val_df = self.val_df[self.val_df['processed'].str.len() > 0]
        self.test_df = self.test_df[self.test_df['processed'].str.len() > 0]
        
        print(f"Po czyszczeniu: train={len(self.train_df)}, val={len(self.val_df)}, test={len(self.test_df)}")
        
        # TF-IDF wektoryzacja
        print("TF-IDF wektoryzacja...")
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        
        # Fit tylko na danych treningowych
        X_train = self.vectorizer.fit_transform(self.train_df['processed'])
        X_val = self.vectorizer.transform(self.val_df['processed'])
        X_test = self.vectorizer.transform(self.test_df['processed'])
        
        y_train = self.train_df['label'].values
        y_val = self.val_df['label'].values
        y_test = self.test_df['label'].values
        
        print(f"Macierze cech: {X_train.shape}, {X_val.shape}, {X_test.shape}")
        print(f"Słownik TF-IDF: {len(self.vectorizer.get_feature_names_out())} cech")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_preprocessing_stats(self):
        """Statystyki przetwarzania"""
        stats = {
            'stop_words_count': len(self.stop_words),
            'vocabulary_size': len(self.vectorizer.get_feature_names_out()) if self.vectorizer else 0,
            'train_samples': len(self.train_df) if hasattr(self, 'train_df') else 0,
            'val_samples': len(self.val_df) if hasattr(self, 'val_df') else 0,
            'test_samples': len(self.test_df) if hasattr(self, 'test_df') else 0
        }
        return stats
    
    def analyze_feature_examples(self, n_examples=10):
        """Analiza przykładowych cech"""
        if self.vectorizer is None:
            print("Brak wytrenowanego vectorizer'a")
            return
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Najwyższe TF-IDF scores
        X_train_dense = self.vectorizer.transform(self.train_df['processed']).toarray()
        mean_scores = np.mean(X_train_dense, axis=0)
        top_indices = np.argsort(mean_scores)[-n_examples:][::-1]

        print(50*'-')
        print(f"Top {n_examples} cech wg średniego TF-IDF:")
        for idx in top_indices:
            print(f"  {feature_names[idx]}: {mean_scores[idx]:.4f}")

def prepare_data(train_path='data/train.txt', val_path='data/valid.txt', test_path='data/test.txt', 
                max_features=5000, ngram_range=(1, 2)):
    """Główna funkcja przygotowania danych"""
    
    preprocessor = TextPreprocessor()
    
    # Wczytanie danych
    train_df, val_df, test_df = preprocessor.load_all_data(train_path, val_path, test_path)
    
    # Przygotowanie cech
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_features(
        max_features=max_features, 
        ngram_range=ngram_range
    )
    
    # Analiza cech
    preprocessor.analyze_feature_examples()
    
    # Statystyki
    stats = preprocessor.get_preprocessing_stats()
    
    return (X_train, X_val, X_test, y_train, y_val, y_test), preprocessor, stats

if __name__ == "__main__":
    main()