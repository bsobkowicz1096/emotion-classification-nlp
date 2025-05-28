# Klasyfikacja Emocji w Tekstach - Projekt NLP

## Przegląd
Model automatycznej klasyfikacji emocji dla firmy monitorującej media społecznościowe. Projekt implementuje pipeline uczenia maszynowego do kategoryzacji tekstów na 6 emocji: **radość**, **smutek**, **złość**, **strach**, **miłość** i **zaskoczenie**.

## Kontekst Biznesowy
Firmy monitorujące media społecznościowe potrzebują zautomatyzowanych narzędzi do analizy tysięcy postów dziennie. Manualna kategoryzacja emocji jest czasochłonna i kosztowna, ograniczając możliwość reakcji w czasie rzeczywistym na zmiany nastrojów konsumentów.

## Wyniki
- **82% dokładności na zbiorze testowym** - przekracza założony próg biznesowy 75%
- Najlepsze wyniki: **smutek** (F1: 0.873), **złość** (F1: 0.846)
- Najtrudniejsze klasy: **zaskoczenie** (F1: 0.673), **miłość** (F1: 0.704)
- Model gotowy do wdrożenia produkcyjnego

## Stack Technologiczny
- **Python 3.12**
- **scikit-learn** - Uczenie maszynowe
- **TF-IDF** - Wektoryzacja tekstu
- **Regresja logistyczna** - Najlepszy model
- **Pandas, NumPy** - Przetwarzanie danych
- **Matplotlib, Seaborn** - Wizualizacja

## Struktura Projektu
```
emotion-classification-nlp/
├── README.md
├── requirements.txt
├── analiza_sentymentu_emotions.ipynb    # Główny notebook z analizą
├── data/
│   ├── train.txt              # Zbiór treningowy
│   ├── val.txt                # Zbiór walidacyjny
│   └── test.txt               # Zbiór testowy
├── src/
│   ├── __init__.py
│   ├── data_exploration.py    # Funkcje eksploracji danych
│   ├── preprocessing.py       # Pipeline przetwarzania tekstu
│   ├── modeling.py           # Trenowanie i porównanie modeli
│   ├── evaluation.py         # Finalna ewaluacja i tuning
│   └── deployment.py         # Wdrożenie modelu do produkcji
└── results/
    ├── confusion_matrix.png
    ├── training_data_exploration.png
    └── emotion_model.pkl      # Zapisany model
```

## Instalacja
1. Sklonuj repozytorium:
```bash
git clone https://github.com/twojusername/emotion-classification-nlp.git
cd emotion-classification-nlp
```

2. Zainstaluj zależności:
```bash
pip install -r requirements.txt
```

## Użycie

Otwórz `analiza_sentymentu_emotions.ipynb` w Jupyter Notebook/Lab dla kompletnej analizy z wizualizacjami i szczegółowymi wyjaśnieniami.

```bash
jupyter notebook analiza_sentymentu_emotions.ipynb
```

## Kluczowe Funkcjonalności

### Inteligentne Przetwarzanie Tekstu
- **Automatyczne wykrywanie stop words** używając heurystyki (słowa występujące w >67% klas emocji)
- **Wektoryzacja TF-IDF** z uni- i bi-gramami
- **Zbalansowane wagi klas** do radzenia sobie z nierównowagą danych

### Porównanie Modeli
Przetestowano 3 algorytmy:
- **Regresja logistyczna** (Zwycięzca: 80.93% dokładności walidacyjnej)
- **Random Forest** (78.00% dokładności, problemy z overfittingiem)
- **Naive Bayes** (78.72% dokładności, najlepsza generalizacja)

### Zaawansowana Analiza
- **Macierz pomyłek** ujawniająca wzorce błędnej klasyfikacji emocji
- **Ważność cech** pokazująca charakterystyczne słowa dla każdej emocji
- **Analiza błędów** identyfikująca wyzwania językowe (ironia, mieszane emocje)

## Zbiór Danych
- **Trening**: 16,000 próbek
- **Walidacja**: 2,000 próbek  
- **Test**: 2,000 próbek
- **Format**: Pary tekst-emocja oddzielone średnikiem
- **Źródło**: [Emotions Dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)

## Kluczowe Obserwacje
1. **Radość często mylona ze smutkiem** - wskazuje na obecność ironii/sarkazmu w mediach społecznościowych
2. **Zaskoczenie najtrudniejsze do klasyfikacji** - z powodu nierównowagi danych (3.6% próbek)
3. **Wulgarny język koreluje z silnymi emocjami** - ważne dla moderacji treści
4. **Kontekst ma znaczenie** - te same słowa mogą wyrażać różne emocje w zależności od użycia

## Zastosowania Biznesowe
- **Zarządzanie kryzysowe** - Szybka detekcja negatywnych nastrojów
- **Monitoring kampanii** - Śledzenie pozytywnych reakcji na marketing
- **Moderacja treści** - Automatyczne filtrowanie oparte na emocjach
- **Insights konsumenckie** - Zrozumienie emocjonalnych reakcji klientów

## Przyszłe Usprawnienia
- Zebranie większej ilości danych dla niedoreprezentowanych emocji (zaskoczenie, miłość)
- Implementacja metod ensemble lub modeli transformer
- Dodanie preprocessing wykrywającego ironię/sarkazm
- Stworzenie kategorii mieszanych emocji

## Metodologia
Projekt realizowany według standardu **CRISP-DM**:
1. **Zrozumienie biznesowe** - Analiza potrzeb firmy monitorującej media
2. **Zrozumienie danych** - Eksploracja 16k tekstów i rozkładu emocji
3. **Przygotowanie danych** - Inteligentne przetwarzanie i wektoryzacja TF-IDF
4. **Modelowanie** - Porównanie algorytmów i wybór najlepszego
5. **Ewaluacja** - Tuning hiperparametrów, testy finalne i analiza wyników
6. **Wdrożenie** - Deployment modelu z interfejsem predykcji

## Licencja
MIT License

## Kontakt
Twoje Imię - twoj.email@example.com
Link do Projektu: [https://github.com/twojusername/emotion-classification-nlp](https://github.com/twojusername/emotion-classification-nlp)