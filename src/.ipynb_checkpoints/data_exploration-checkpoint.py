#!/usr/bin/env python3
"""
Eksploracja danych - analiza struktury i rozkładu
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

def load_txt_data(file_path):
    """Wczytanie danych z pliku TXT"""
    texts = []
    labels = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:  # pomijamy puste linie
                if ';' in line:
                    text, label = line.rsplit(';', 1)  # split od końca
                    texts.append(text)
                    labels.append(label)
    
    return pd.DataFrame({'text': texts, 'label': labels})

def explore_data_structure(df):
    """Podstawowa eksploracja struktury danych"""
    print(f"Liczba próbek: {len(df)}")
    print(f"Kolumny: {df.columns.tolist()}")
    
    # Sprawdzenie brakujących wartości
    print(f"Brakujące wartości: {df.isnull().sum().sum()}")
    
    # Unikalne etykiety
    print(f"Unikalne etykiety: {df['label'].unique()}")
    print(f"Liczba klas: {df['label'].nunique()}")
    
    return df

def analyze_label_distribution(df):
    """Analiza rozkładu etykiet"""
    label_counts = df['label'].value_counts()
    print(f"\nRozkład etykiet:")
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{label}: {count} ({percentage:.1f}%)")
    
    return label_counts

def analyze_text_statistics(df):
    """Analiza statystyk tekstów"""
    # Długość tekstów (znaki)
    char_lengths = df['text'].str.len()
    
    # Liczba słów
    word_counts = df['text'].str.split().str.len()
    
    print(f"\nStatystyki tekstów:")
    print(f"Długość (znaki) - średnia: {char_lengths.mean():.1f}, mediana: {char_lengths.median():.1f}")
    print(f"Liczba słów - średnia: {word_counts.mean():.1f}, mediana: {word_counts.median():.1f}")
    print(f"Najkrótszy tekst: {char_lengths.min()} znaków")
    print(f"Najdłuższy tekst: {char_lengths.max()} znaków")
    
    return char_lengths, word_counts


def find_generic_words(df, threshold):
    """Znajdź słowa występujące w >threshold emocji (niecharakterystyczne)"""
    emotion_words = {}
    for emotion in df['label'].unique():
        texts = ' '.join(df[df['label'] == emotion]['text']).lower()
        words = set(re.findall(r'\b[a-zA-Z]{3,}\b', texts))
        emotion_words[emotion] = words
    
    # Słowa występujące w większości emocji
    all_words = set.union(*emotion_words.values())
    generic_words = []
    
    for word in all_words:
        emotion_count = sum(1 for words in emotion_words.values() if word in words)
        if emotion_count / len(emotion_words) > threshold:
            generic_words.append(word)
    
    return sorted(generic_words)

def find_most_common_words_by_emotion(df):
    """Analiza najczęstszych słów per emocja (top 5)"""
    print("\nNajczęstsze słowa per emocja (top 5):")
    
    # Automatyczne wykrycie słów niecharakterystycznych
    generic_words = find_generic_words(df, threshold=0.66)

    stop_words = set(generic_words)
    
    for emotion in df['label'].unique():
        emotion_texts = df[df['label'] == emotion]['text']
        
        # Połączenie wszystkich tekstów dla danej emocji
        all_text = ' '.join(emotion_texts).lower()
        
        # Podstawowe czyszczenie i tokenizacja
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text)
        
        # Filtrowanie stop words
        filtered_words = [word for word in words if word not in stop_words]
        
        # Najczęstsze słowa
        common_words = Counter(filtered_words).most_common(5)
        
        print(f"{emotion}: {[word for word, count in common_words]}")

def explore_training_data(train_path='data/train.txt', save_plots=True):
    """Główna funkcja eksploracyjna zwracająca wyniki"""
    print("=== EKSPLORACJA DANYCH TRENINGOWYCH ===")
    
    try:
        # Wczytanie danych
        train_df = load_txt_data(train_path)
        
        # Podstawowa eksploracja
        explore_data_structure(train_df)
        
        # Analiza rozkładu etykiet
        label_counts = analyze_label_distribution(train_df)
        
        # Statystyki tekstów
        char_lengths, word_counts = analyze_text_statistics(train_df)
        
        # Wizualizacje
        if save_plots:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Rozkład klas
            label_counts.plot(kind='bar', ax=axes[0], color='skyblue', alpha=0.8)
            axes[0].set_title('Rozkład emocji')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(True, alpha=0.3)
            
            # Rozkład długości tekstów
            char_lengths.hist(bins=30, ax=axes[1], alpha=0.8, color='skyblue')
            axes[1].set_title('Rozkład długości tekstów')
            axes[1].set_xlabel('Liczba znaków')
            axes[1].set_ylabel('Częstość')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('results/training_data_exploration.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Analiza słów per emocja (z automatycznym wykrywaniem stop words)
        find_most_common_words_by_emotion(train_df)
        
        # Osobno dla wyników
        generic_words = find_generic_words(train_df, threshold=0.66)
        
        # Przygotowanie wyników
        results = {
            'train_size': len(train_df),
            'emotions': train_df['label'].unique().tolist(),
            'label_distribution': label_counts.to_dict(),
            'avg_text_length_chars': char_lengths.mean(),
            'avg_text_length_words': word_counts.mean(),
            'median_text_length': char_lengths.median(),
            'min_max_length': (char_lengths.min(), char_lengths.max()),
            'stop_words_count': len(generic_words),
            'detected_stop_words': sorted(list(generic_words))
        }
        
        return train_df, results
        
    except FileNotFoundError as e:
        print(f"Błąd: Nie znaleziono pliku {e}")
        return None, None
    except Exception as e:
        print(f"Błąd podczas eksploracji: {e}")
        return None, None

def main():
    """Funkcja do uruchomienia jako skrypt"""
    train_df, results = explore_training_data()
    
if __name__ == "__main__":
    main()