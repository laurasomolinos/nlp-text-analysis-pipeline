# NLP Text Analysis Pipeline

This project implements a complete Natural Language Processing (NLP) pipeline for text preprocessing, feature extraction, classification, information extraction, and summarisation.

Procesamiento de Lenguaje Natural - Grado Ciencia de Datos e Inteligencia Artificial (2024)

## Features

- Text preprocessing (cleaning, stopwords, normalization)
- Word2Vec embeddings
- Subreddit classification (SVM)
- Sentiment analysis (Naive Bayes)
- Regex-based information extraction
- Text summarisation (TF-IDF)
- Text similarity (cosine distance)

## Technologies

- Python
- NLTK
- Gensim
- Scikit-learn
- Pandas
- NumPy

## Project Structure
```text
nlp-text-analysis-pipeline
├── core.py
├── testing.ipynb
├── README.md
├── requirements.txt
└── data/
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Download NLTK resources:
```bash
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Dataset

The dataset is not included in this repository.

To run the project, place the required data files inside the data/ folder.

## Notes

This project was developed for academic purposes.
